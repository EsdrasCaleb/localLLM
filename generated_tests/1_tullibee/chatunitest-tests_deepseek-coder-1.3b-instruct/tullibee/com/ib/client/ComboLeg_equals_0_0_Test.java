package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ComboLeg_equals_0_0_Test {

    @Test
    public void testEquals() {
        ComboLeg leg1 = new ComboLeg(1, 2, "action1", "exchange1", ComboLeg.OPEN);
        ComboLeg leg2 = new ComboLeg(1, 2, "action1", "exchange1", ComboLeg.OPEN);
        ComboLeg leg3 = new ComboLeg(3, 4, "action2", "exchange2", ComboLeg.CLOSE);
        assertTrue(leg1.equals(leg2));
        assertFalse(leg1.equals(leg3));
    }
}
