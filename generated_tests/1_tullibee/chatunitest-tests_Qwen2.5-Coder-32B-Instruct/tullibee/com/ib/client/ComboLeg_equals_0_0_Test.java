package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ComboLeg_equals_0_0_Test {

    @Mock
    private Util mockUtil;

    private ComboLeg comboLeg;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        comboLeg = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN, 0, "LOC");
    }

    @Test
    public void testEquals_SameInstance_ReturnsTrue() {
        assertTrue(comboLeg.equals(comboLeg));
    }

    @Test
    public void testEquals_NullObject_ReturnsFalse() {
        assertFalse(comboLeg.equals(null));
    }

    @Test
    public void testEquals_AllFieldsEqual_ReturnsTrue() {
        ComboLeg otherComboLeg = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN, 0, "LOC");
        assertTrue(comboLeg.equals(otherComboLeg));
    }

    @Test
    public void testEquals_DifferentConId_ReturnsFalse() {
        ComboLeg otherComboLeg = new ComboLeg(2, 2, "BUY", "NYSE", ComboLeg.OPEN, 0, "LOC");
        assertFalse(comboLeg.equals(otherComboLeg));
    }

    @Test
    public void testEquals_DifferentRatio_ReturnsFalse() {
        ComboLeg otherComboLeg = new ComboLeg(1, 3, "BUY", "NYSE", ComboLeg.OPEN, 0, "LOC");
        assertFalse(comboLeg.equals(otherComboLeg));
    }

    @Test
    public void testEquals_DifferentOpenClose_ReturnsFalse() {
        ComboLeg otherComboLeg = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.CLOSE, 0, "LOC");
        assertFalse(comboLeg.equals(otherComboLeg));
    }

    @Test
    public void testEquals_DifferentShortSaleSlot_ReturnsFalse() {
        ComboLeg otherComboLeg = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN, 1, "LOC");
        assertFalse(comboLeg.equals(otherComboLeg));
    }
}
