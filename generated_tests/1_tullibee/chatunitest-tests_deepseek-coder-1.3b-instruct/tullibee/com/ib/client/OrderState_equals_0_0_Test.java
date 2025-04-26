package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class OrderState_equals_0_0_Test {

    @Test
    public void testEquals() {
        OrderState state1 = new OrderState("status1", "initMargin1", "maintMargin1", "equityWithLoan1", 1.0, 2.0, 3.0, "commissionCurrency1", "warningText1");
        OrderState state2 = new OrderState("status2", "initMargin2", "maintMargin2", "equityWithLoan2", 4.0, 5.0, 6.0, "commissionCurrency2", "warningText2");
        // Test equal objects
        assertTrue(state1.equals(state2));
        // Test different objects
        OrderState state3 = new OrderState("status3", "initMargin3", "maintMargin3", "equityWithLoan3", 7.0, 8.0, 9.0, "commissionCurrency3", "warningText3");
        assertFalse(state1.equals(state3));
        // Test null object
        assertFalse(state1.equals(null));
        // Test different class
        assertFalse(state1.equals(new Object()));
    }
}
