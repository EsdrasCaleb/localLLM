package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ComboLeg_equals_0_1_Test {

    @Test
    public void testEquals() {
        ComboLeg leg1 = new ComboLeg(1, 0, "open", "open", 0, 0, "location");
        ComboLeg leg2 = new ComboLeg(1, 0, "open", "open", 0, 0, "location");
        ComboLeg leg3 = new ComboLeg(1, 1, "open", "open", 0, 0, "location");
        assertEquals(leg1, leg2);
        assertEquals(leg1, leg3);
    }
}
