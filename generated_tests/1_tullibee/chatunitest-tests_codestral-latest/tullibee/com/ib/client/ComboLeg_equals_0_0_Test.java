package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ComboLeg_equals_0_0_Test {

    private ComboLeg comboLeg1;

    private ComboLeg comboLeg2;

    @BeforeEach
    void setUp() {
        comboLeg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN, 0, "Location1");
        comboLeg2 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN, 0, "Location1");
    }

    @Test
    void testEquals_SameObject() {
        assertTrue(comboLeg1.equals(comboLeg1));
    }

    @Test
    void testEquals_NullObject() {
        assertFalse(comboLeg1.equals(null));
    }

    @Test
    void testEquals_DifferentConId() {
        comboLeg2.m_conId = 2;
        assertFalse(comboLeg1.equals(comboLeg2));
    }

    @Test
    void testEquals_DifferentRatio() {
        comboLeg2.m_ratio = 3;
        assertFalse(comboLeg1.equals(comboLeg2));
    }

    @Test
    void testEquals_DifferentOpenClose() {
        comboLeg2.m_openClose = ComboLeg.CLOSE;
        assertFalse(comboLeg1.equals(comboLeg2));
    }

    @Test
    void testEquals_DifferentShortSaleSlot() {
        comboLeg2.m_shortSaleSlot = 1;
        assertFalse(comboLeg1.equals(comboLeg2));
    }

    @Test
    void testEquals_DifferentAction() {
        comboLeg2.m_action = "SELL";
        assertFalse(comboLeg1.equals(comboLeg2));
    }

    @Test
    void testEquals_DifferentExchange() {
        comboLeg2.m_exchange = "NASDAQ";
        assertFalse(comboLeg1.equals(comboLeg2));
    }

    @Test
    void testEquals_DifferentDesignatedLocation() {
        comboLeg2.m_designatedLocation = "Location2";
        assertFalse(comboLeg1.equals(comboLeg2));
    }

    @Test
    void testEquals_EqualObjects() {
        assertTrue(comboLeg1.equals(comboLeg2));
    }
}
