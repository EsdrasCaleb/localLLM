package com.ib.client;

import static org.junit.Assert.assertEquals;
import org.junit.Test;
import java.util.Objects;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ComboLeg_equals_0_0_Test {

    @Test
    public void testEquals() {
        // Test case 1: Both objects are the same
        ComboLeg l1 = new ComboLeg(1, 1, "Open", "XYZ", 1, 0, "Central");
        ComboLeg l2 = new ComboLeg(1, 1, "Open", "XYZ", 1, 0, "Central");
        assertEquals(l1, l2);
        // Test case 2: One object is null
        ComboLeg l3 = null;
        assertEquals(l3, l1);
        // Test case 3: One object has different conId
        ComboLeg l4 = new ComboLeg(1, 1, "Open", "XYZ", 1, 0, "Central");
        l4.m_conId = 2;
        assertEquals(l4, l1);
        // Test case 4: One object has different ratio
        ComboLeg l5 = new ComboLeg(1, 1, "Open", "XYZ", 1, 1, "Central");
        l5.m_ratio = 2;
        assertEquals(l5, l1);
        // Test case 5: One object has different openClose status
        ComboLeg l6 = new ComboLeg(1, 1, "Open", "XYZ", 1, 0, "Central");
        l6.m_openClose = 2;
        assertEquals(l6, l1);
        // Test case 6: One object has different shortSaleSlot
        ComboLeg l7 = new ComboLeg(1, 1, "Open", "XYZ", 1, 0, "Central");
        l7.m_shortSaleSlot = 2;
        assertEquals(l7, l1);
        // Test case 7: One object has different action
        ComboLeg l8 = new ComboLeg(1, 1, "Open", "XYZ", 1, 0, "Central");
        l8.m_action = "Buy";
        assertEquals(l8, l1);
        // Test case 8: One object has different exchange
        ComboLeg l9 = new ComboLeg(1, 1, "Open", "XYZ", 1, 0, "Central");
        l9.m_exchange = "ABC";
        assertEquals(l9, l1);
        // Test case 9: One object has different designatedLocation
        ComboLeg l10 = new ComboLeg(1, 1, "Open", "XYZ", 1, 0, "Central");
        l10.m_designatedLocation = "DEF";
        assertEquals(l10, l1);
    }
}
