package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ComboLeg_equals_0_1_Test {

    @Test
    public void testEquals_SameInstance_ReturnsTrue() {
        ComboLeg comboLeg = new ComboLeg(1, 2, "action", "exchange", 3, 4, "location");
        assertTrue(comboLeg.equals(comboLeg));
    }

    @Test
    public void testEquals_NullObject_ReturnsFalse() {
        ComboLeg comboLeg = new ComboLeg(1, 2, "action", "exchange", 3, 4, "location");
        assertFalse(comboLeg.equals(null));
    }

    @Test
    public void testEquals_DifferentInstances_ReturnsFalse() {
        ComboLeg comboLeg1 = new ComboLeg(1, 2, "action", "exchange", 3, 4, "location");
        ComboLeg comboLeg2 = new ComboLeg(1, 2, "action", "exchange", 3, 4, "location");
        assertFalse(comboLeg1.equals(comboLeg2));
    }

    @Test
    public void testEquals_MismatchedFields_ReturnsFalse() {
        ComboLeg comboLeg1 = new ComboLeg(1, 2, "action", "exchange", 3, 4, "location");
        ComboLeg comboLeg2 = new ComboLeg(1, 3, "differentAction", "exchange", 3, 4, "location");
        assertFalse(comboLeg1.equals(comboLeg2));
    }

    @Test
    public void testEquals_MismatchedNonFinalFields_ReturnsFalse() {
        ComboLeg comboLeg1 = new ComboLeg(1, 2, "action", "exchange", 3, 4, "location");
        ComboLeg comboLeg2 = new ComboLeg(1, 2, "action", "exchange", 3, 4, "location");
        comboLeg2.m_ratio = 5;
        assertFalse(comboLeg1.equals(comboLeg2));
    }
}
