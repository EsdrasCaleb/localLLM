package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class OrderState_equals_0_0_Test {

    @Mock
    private Util utilMock;

    private OrderState orderState;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        orderState = new OrderState("status", "initMargin", "maintMargin", "equityWithLoan", 1.0, 2.0, 3.0, "currency", "warning");
    }

    @Test
    public void testEquals_SameInstance_ReturnsTrue() {
        assertTrue(orderState.equals(orderState));
    }

    @Test
    public void testEquals_NullObject_ReturnsFalse() {
        assertFalse(orderState.equals(null));
    }

    @Test
    public void testEquals_SameValues_ReturnsTrue() {
        OrderState other = new OrderState("status", "initMargin", "maintMargin", "equityWithLoan", 1.0, 2.0, 3.0, "currency", "warning");
        assertTrue(orderState.equals(other));
    }

    @Test
    public void testEquals_DifferentInitMargin_ReturnsFalse() {
        OrderState other = new OrderState("status", "differentInitMargin", "maintMargin", "equityWithLoan", 1.0, 2.0, 3.0, "currency", "warning");
        assertFalse(orderState.equals(other));
    }

    @Test
    public void testEquals_DifferentMaintMargin_ReturnsFalse() {
        OrderState other = new OrderState("status", "initMargin", "differentMaintMargin", "equityWithLoan", 1.0, 2.0, 3.0, "currency", "warning");
        assertFalse(orderState.equals(other));
    }

    @Test
    public void testEquals_DifferentEquityWithLoan_ReturnsFalse() {
        OrderState other = new OrderState("status", "initMargin", "maintMargin", "differentEquityWithLoan", 1.0, 2.0, 3.0, "currency", "warning");
        assertFalse(orderState.equals(other));
    }

    @Test
    public void testEquals_DifferentCommissionCurrency_ReturnsFalse() {
        OrderState other = new OrderState("status", "initMargin", "maintMargin", "equityWithLoan", 1.0, 2.0, 3.0, "differentCurrency", "warning");
        assertFalse(orderState.equals(other));
    }

    @Test
    public void testEquals_DifferentCommission_ReturnsFalse() {
        OrderState other = new OrderState("status", "initMargin", "maintMargin", "equityWithLoan", 4.0, 2.0, 3.0, "currency", "warning");
        assertFalse(orderState.equals(other));
    }

    @Test
    public void testEquals_DifferentMinCommission_ReturnsFalse() {
        OrderState other = new OrderState("status", "initMargin", "maintMargin", "equityWithLoan", 1.0, 5.0, 3.0, "currency", "warning");
        assertFalse(orderState.equals(other));
    }

    @Test
    public void testEquals_DifferentMaxCommission_ReturnsFalse() {
        OrderState other = new OrderState("status", "initMargin", "maintMargin", "equityWithLoan", 1.0, 2.0, 6.0, "currency", "warning");
        assertFalse(orderState.equals(other));
    }
}
