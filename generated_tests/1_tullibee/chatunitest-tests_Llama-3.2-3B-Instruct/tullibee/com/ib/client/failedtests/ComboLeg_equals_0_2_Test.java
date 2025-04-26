package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ComboLeg_equals_0_2_Test {

    @InjectMocks
    ComboLeg comboLeg;

    @Test
    public void testEquals_SameInstance() {
        assertEquals(true, comboLeg.equals(comboLeg));
    }

    @Test
    public void testEquals_Null() {
        assertFalse(comboLeg.equals(null));
    }

    @Test
    public void testEquals_DifferentTypes() {
        Object obj = new Object();
        assertFalse(comboLeg.equals(obj));
    }

    @Test
    public void testEquals_DifferentAttributes() {
        comboLeg.m_conId = 1;
        comboLeg.m_ratio = 2;
        comboLeg.m_openClose = 3;
        comboLeg.m_shortSaleSlot = 4;
        comboLeg.m_designatedLocation = "Test";
        ComboLeg otherComboLeg = new ComboLeg(1, 2, "Test", "Test", 3, 4, "Test");
        assertFalse(comboLeg.equals(otherComboLeg));
    }

    @Test
    public void testEquals_SameAttributes() {
        comboLeg.m_conId = 1;
        comboLeg.m_ratio = 2;
        comboLeg.m_openClose = 3;
        comboLeg.m_shortSaleSlot = 4;
        comboLeg.m_designatedLocation = "Test";
        ComboLeg otherComboLeg = new ComboLeg(1, 2, "Test", "Test", 3, 4, "Test");
        assertTrue(comboLeg.equals(otherComboLeg));
    }

    @Test
    public void testEquals_MismatchedAction() {
        comboLeg.m_action = "Test";
        comboLeg.m_exchange = "Test";
        comboLeg.m_designatedLocation = "Test";
        ComboLeg otherComboLeg = new ComboLeg(1, 2, "Different", "Test", 3, 4, "Test");
        assertFalse(comboLeg.equals(otherComboLeg));
    }

    @Test
    public void testEquals_MismatchedExchange() {
        comboLeg.m_action = "Test";
        comboLeg.m_exchange = "Test";
        comboLeg.m_designatedLocation = "Test";
        ComboLeg otherComboLeg = new ComboLeg(1, 2, "Test", "Different", 3, 4, "Test");
        assertFalse(comboLeg.equals(otherComboLeg));
    }

    @Test
    public void testEquals_MismatchedDesignatedLocation() {
        comboLeg.m_action = "Test";
        comboLeg.m_exchange = "Test";
        comboLeg.m_designatedLocation = "Test";
        ComboLeg otherComboLeg = new ComboLeg(1, 2, "Test", "Test", 3, 4, "Different");
        assertFalse(comboLeg.equals(otherComboLeg));
    }
}
