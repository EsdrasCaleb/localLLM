package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class ComboLeg_equals_0_4_Test {

    // Test class
    @Test
    public void testEquals() {
        ComboLeg l_this = new ComboLeg(1, 1, "B", "SMART", ComboLeg.OPEN);
        ComboLeg l_other = new ComboLeg(1, 1, "B", "SMART", ComboLeg.OPEN);
        assertTrue(l_this.equals(l_other));
        l_other = new ComboLeg(2, 1, "B", "SMART", ComboLeg.OPEN);
        assertFalse(l_this.equals(l_other));
        l_other = new ComboLeg(1, 2, "B", "SMART", ComboLeg.OPEN);
        assertFalse(l_this.equals(l_other));
        l_other = new ComboLeg(1, 1, "B", "SMART", ComboLeg.CLOSE);
        assertFalse(l_this.equals(l_other));
        l_other = new ComboLeg(1, 1, "C", "SMART", ComboLeg.OPEN);
        assertFalse(l_this.equals(l_other));
        l_other = new ComboLeg(1, 1, "B", "SMART", ComboLeg.UNKNOWN);
        assertFalse(l_this.equals(l_other));
        l_other = new ComboLeg(1, 1, "B", "SMART", ComboLeg.SAME);
        assertFalse(l_this.equals(l_other));
    }
}
