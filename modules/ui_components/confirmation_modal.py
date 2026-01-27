"""
Reusable confirmation modal for Gradio applications.
Provides a styled confirmation dialog that matches the Gradio theme.

Usage:
    1. Import CONFIRMATION_MODAL_CSS, CONFIRMATION_MODAL_HEAD, CONFIRMATION_MODAL_HTML
    2. Pass css and head to demo.launch(css=CONFIRMATION_MODAL_CSS, head=CONFIRMATION_MODAL_HEAD, ...)
    3. Add CONFIRMATION_MODAL_HTML via gr.HTML() in your UI
    4. Add a hidden confirm_trigger textbox with elem_id="confirm-trigger"
    5. Use show_confirmation_modal_js() for button click handlers
    6. In your callback, filter empty actions: if not action or not action.strip(): return gr.update(), gr.update()

Note: The callback will be triggered twice per button click (once with action, once with empty string).
      This is expected behavior. Simply filter out empty calls in your callback function.
"""

CONFIRMATION_MODAL_CSS = """
/* Confirmation Modal Styles */
#delete-modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(4px);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}
#delete-modal-overlay.show {
  display: flex;
}
#delete-modal-overlay .modal-box {
  background: var(--background-fill-primary);
  border: 1px solid var(--border-color-primary);
  color: var(--body-text-color);
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  min-width: 400px;
  max-width: 500px;
  animation: modalSlideIn 0.2s ease-out;
}
@keyframes modalSlideIn {
  from {
    transform: translateY(-20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}
#delete-modal-overlay h3 {
  margin: 0 0 1rem 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--body-text-color);
}
#delete-modal-overlay p {
  margin: 0 0 1.5rem 0;
  color: var(--body-text-color-subdued);
  line-height: 1.5;
}
#delete-modal-overlay .modal-actions {
  display: flex;
  gap: 0.75rem;
  justify-content: flex-end;
}
#delete-modal-overlay .modal-btn {
  padding: 0.625rem 1.5rem;
  border-radius: 6px;
  border: 1px solid var(--border-color-primary);
  background: var(--button-secondary-background-fill);
  color: var(--button-secondary-text-color);
  cursor: pointer;
  font-size: 0.95rem;
  font-weight: 500;
  font-family: inherit;
  transition: all 0.15s ease;
  height: 42px;
  min-width: 100px;
  line-height: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  box-sizing: border-box;
}
#delete-modal-overlay .modal-btn:hover {
  background: var(--button-secondary-background-fill-hover);
  border-color: var(--border-color-accent);
}
#delete-modal-overlay .modal-btn.danger {
  background: var(--button-cancel-background-fill);
  color: var(--button-cancel-text-color);
  border-color: var(--button-cancel-border-color);
}
#delete-modal-overlay .modal-btn.danger:hover {
  background: var(--button-cancel-background-fill-hover);
  border-color: var(--button-cancel-border-color-hover);
}
"""

CONFIRMATION_MODAL_HEAD = """
<script>
  function confirmModalAction(action, button) {
    console.log('confirmModalAction called with:', action);

    const overlay = document.getElementById('delete-modal-overlay');
    if (overlay) {
      overlay.classList.remove('show');
    }

    // Get context from button's data attribute
    const context = button ? button.getAttribute('data-context') || '' : '';
    const prefixedAction = context + action;
    console.log('Prefixed action:', prefixedAction);

    // Try to find the trigger - Gradio 5+ has different DOM structure
    function findTrigger() {
      // Try various selectors
      let trigger = document.querySelector('#confirm-trigger textarea');
      if (trigger) return trigger;

      trigger = document.querySelector('#confirm-trigger input[type="text"]');
      if (trigger) return trigger;

      trigger = document.querySelector('#confirm-trigger input');
      if (trigger) return trigger;

      // Try finding by data attribute or class
      const container = document.querySelector('[id="confirm-trigger"]');
      if (container) {
        trigger = container.querySelector('textarea, input');
        if (trigger) return trigger;
      }

      // Try searching in all textboxes for one that's hidden
      const allInputs = document.querySelectorAll('textarea, input[type="text"]');
      for (let input of allInputs) {
        const parent = input.closest('[id="confirm-trigger"]');
        if (parent) return input;
      }

      return null;
    }

    const trigger = findTrigger();

    if (trigger) {
      const newValue = prefixedAction + '_' + Date.now();
      console.log('Setting trigger value to:', newValue);
      trigger.value = newValue;

      trigger.dispatchEvent(new Event('input', { bubbles: true }));
      trigger.dispatchEvent(new Event('change', { bubbles: true }));
      const evt = new InputEvent('input', { bubbles: true, cancelable: true });
      trigger.dispatchEvent(evt);

      console.log('Events dispatched, trigger value is now:', trigger.value);
    } else {
      console.error('Could not find confirm-trigger element');
      console.log('Available elements with confirm-trigger id:', document.querySelectorAll('[id*="confirm-trigger"]'));
    }
  }

  window.addEventListener('DOMContentLoaded', () => {
    console.log('DOMContentLoaded - setting up modal listeners');
    const overlay = document.getElementById('delete-modal-overlay');
    if (!overlay) {
      console.error('Modal overlay not found!');
      return;
    }
    overlay.addEventListener('click', function(e) {
      if (e.target === this) {
        confirmModalAction('cancel');
      }
    });
  });

  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      const overlay = document.getElementById('delete-modal-overlay');
      if (overlay && overlay.classList.contains('show')) {
        confirmModalAction('cancel');
      }
    }
  });
</script>
"""

CONFIRMATION_MODAL_HTML = """
<div id="delete-modal-overlay">
  <div class="modal-box">
    <h3 id="confirm-modal-title">Confirm Action</h3>
    <p id="confirm-modal-message">Are you sure you want to continue?</p>
    <div class="modal-actions">
      <button class="modal-btn" id="confirm-modal-cancel-btn" onclick="confirmModalAction('cancel', this)">Cancel</button>
      <button class="modal-btn danger" id="confirm-modal-action-btn" onclick="confirmModalAction('confirm', this)">Confirm</button>
    </div>
  </div>
</div>
"""


def show_confirmation_modal_js(title, message, confirm_button_text="Delete", context=""):
    """
    Generate JavaScript function to show the confirmation modal.

    Args:
        title: Modal title (e.g., "Delete Sample?")
        message: Modal message (e.g., "This will permanently delete the sample.")
        confirm_button_text: Text for the confirm button (default: "Delete")
        context: Context prefix for action (e.g., "sample_", "output_", "cache_")

    Returns:
        JavaScript function string to use in Gradio's .click(js=...) parameter
    """
    return f"""
    () => {{
        const overlay = document.getElementById('delete-modal-overlay');
        if (!overlay) return '';
        const titleEl = document.getElementById('confirm-modal-title');
        const messageEl = document.getElementById('confirm-modal-message');
        const actionBtn = document.getElementById('confirm-modal-action-btn');
        const cancelBtn = document.getElementById('confirm-modal-cancel-btn');

        if (titleEl) titleEl.textContent = {title!r};
        if (messageEl) messageEl.textContent = {message!r};
        if (actionBtn) {{
            actionBtn.textContent = {confirm_button_text!r};
            actionBtn.setAttribute('data-context', {context!r});
        }}
        if (cancelBtn) {{
            cancelBtn.setAttribute('data-context', {context!r});
        }}

        overlay.classList.add('show');
        return '';
    }}
    """


def create_confirmation_workflow(button, confirm_callback, cancel_callback=None,
                                 title="Confirm Action", message="Are you sure?",
                                 confirm_button_text="Confirm"):
    """
    Helper to set up a complete confirmation workflow with a button.

    Args:
        button: Gradio Button component
        confirm_callback: Function to call when user confirms (receives inputs)
        cancel_callback: Optional function to call when user cancels
        title: Modal title
        message: Modal message
        confirm_button_text: Text for confirm button

    Returns:
        tuple: (confirm_trigger component, js_function)
        You need to add confirm_trigger to your UI (can be hidden)
    """
    import gradio as gr

    # Create hidden trigger
    confirm_trigger = gr.Textbox(visible=False, elem_id="confirm-trigger")

    # JavaScript to show modal
    js_func = show_confirmation_modal_js(title, message, confirm_button_text)

    return confirm_trigger, js_func
