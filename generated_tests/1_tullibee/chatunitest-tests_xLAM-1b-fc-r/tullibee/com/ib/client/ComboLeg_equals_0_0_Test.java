package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ComboLeg_equals_0_0_Test {

    @Test
    void equalsTest() {
        ComboLeg leg1 = new ComboLeg(1, 2, "action1", "exchange1", 3);
        ComboLeg leg2 = new ComboLeg(1, 2, "action1", "exchange1", 3);
        ComboLeg leg3 = new ComboLeg(4, 5, "action2", "exchange2", 6);
        assertTrue(leg1.equals(leg2));
        assertFalse(leg1.equals(leg3));
    }
}
