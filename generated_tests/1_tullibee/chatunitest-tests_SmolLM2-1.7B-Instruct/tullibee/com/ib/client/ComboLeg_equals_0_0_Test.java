package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ComboLeg_equals_0_0_Test {

    @Mock
    private ComboLeg mockComboLeg;

    @InjectMocks
    private ComboLeg underTest;

    @Test
    public void testEquals_WithSameObject_ReturnsTrue() {
        ComboLeg sameComboLeg = new ComboLeg(1, 1, "action1", "exchange1", 1, 1, "designatedLocation1");
        assertEquals(true, underTest.equals(sameComboLeg));
    }

    @Test
    public void testEquals_WithNullObject_ReturnsFalse() {
        ComboLeg nullComboLeg = null;
        assertFalse(underTest.equals(nullComboLeg));
    }

    @Test
    public void testEquals_WithDifferentObject_ReturnsFalse() {
        ComboLeg differentComboLeg = new ComboLeg(1, 1, "action2", "exchange2", 1, 1, "designatedLocation2");
        assertFalse(underTest.equals(differentComboLeg));
    }

    @Test
    public void testEquals_WithDifferentAction_ReturnsFalse() {
        ComboLeg sameComboLeg = new ComboLeg(1, 1, "action1", "exchange1", 1, 1, "designatedLocation1");
        ComboLeg differentComboLeg = new ComboLeg(1, 1, "action2", "exchange2", 1, 1, "designatedLocation2");
        assertFalse(underTest.equals(differentComboLeg));
    }

    @Test
    public void testEquals_WithDifferentExchange_ReturnsFalse() {
        ComboLeg sameComboLeg = new ComboLeg(1, 1, "action1", "exchange1", 1, 1, "designatedLocation1");
        ComboLeg differentComboLeg = new ComboLeg(1, 1, "action1", "exchange2", 1, 1, "designatedLocation2");
        assertFalse(underTest.equals(differentComboLeg));
    }

    @Test
    public void testEquals_WithDifferentShortSaleSlot_ReturnsFalse() {
        ComboLeg sameComboLeg = new ComboLeg(1, 1, "action1", "exchange1", 1, 1, "designatedLocation1");
        ComboLeg differentComboLeg = new ComboLeg(1, 1, "action1", "exchange1", 1, 2, "designatedLocation2");
        assertFalse(underTest.equals(differentComboLeg));
    }

    @Test
    public void testEquals_WithDifferentDesignatedLocation_ReturnsFalse() {
        ComboLeg sameComboLeg = new ComboLeg(1, 1, "action1", "exchange1", 1, 1, "designatedLocation1");
        ComboLeg differentComboLeg = new ComboLeg(1, 1, "action1", "exchange1", 1, 1, "designatedLocation2");
        assertFalse(underTest.equals(differentComboLeg));
    }

    @Test
    public void testEquals_WithDifferentRatio_ReturnsFalse() {
        ComboLeg sameComboLeg = new ComboLeg(1, 1, "action1", "exchange1", 1, 1, "designatedLocation1");
        ComboLeg differentComboLeg = new ComboLeg(1, 1, "action1", "exchange1", 2, 1, "designatedLocation2");
        assertFalse(underTest.equals(differentComboLeg));
    }
}
