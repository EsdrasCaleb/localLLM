package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ComboLeg_equals_0_2_Test {

    @Test
    void testEquals_SameObject() {
        ComboLeg comboLeg1 = new ComboLeg(1, 2, "BUY", "NYSE", 0, 0, "A");
        ComboLeg comboLeg2 = comboLeg1;
        assertEquals(comboLeg1, comboLeg2);
    }

    @Test
    void testEquals_DifferentObject() {
        ComboLeg comboLeg1 = new ComboLeg(1, 2, "BUY", "NYSE", 0, 0, "A");
        ComboLeg comboLeg2 = new ComboLeg(1, 3, "SELL", "NASDAQ", 0, 0, "B");
        assertNotEquals(comboLeg1, comboLeg2);
    }
}
