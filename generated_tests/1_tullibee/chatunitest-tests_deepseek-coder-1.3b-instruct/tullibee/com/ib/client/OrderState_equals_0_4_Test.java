package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class OrderState_equals_0_4_Test {

    @Test
    void testEquals() {
        OrderState state1 = new OrderState("status1", "initMargin1", "maintMargin1", "equityWithLoan1", 1.1, 1.2, 1.3, "commissionCurrency1", "warningText1");
        OrderState state2 = new OrderState("status2", "initMargin2", "maintMargin2", "equityWithLoan2", 2.1, 2.2, 2.3, "commissionCurrency2", "warningText2");
        OrderState state3 = new OrderState("status3", "initMargin3", "maintMargin3", "equityWithLoan3", 3.1, 3.2, 3.3, "commissionCurrency3", "warningText3");
        assertTrue(state1.equals(state2));
        assertFalse(state1.equals(state3));
    }
}
