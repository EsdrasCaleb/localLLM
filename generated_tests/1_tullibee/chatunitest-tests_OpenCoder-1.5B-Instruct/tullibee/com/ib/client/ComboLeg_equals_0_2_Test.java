package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ComboLeg_equals_0_2_Test {

    @Test
    public void testEquals() {
        ComboLeg comboLeg1 = new ComboLeg(123, 456, "BUY", "NYMEX", ComboLeg.OPEN, 1, "NY");
        ComboLeg comboLeg2 = new ComboLeg(123, 456, "BUY", "NYMEX", ComboLeg.OPEN, 1, "NY");
        ComboLeg comboLeg3 = new ComboLeg(789, 123, "SELL", "NASDAQ", ComboLeg.CLOSE, 2, "CA");
        // Should pass
        assertTrue(comboLeg1.equals(comboLeg2));
        // Should fail
        assertFalse(comboLeg1.equals(comboLeg3));
    }
}
