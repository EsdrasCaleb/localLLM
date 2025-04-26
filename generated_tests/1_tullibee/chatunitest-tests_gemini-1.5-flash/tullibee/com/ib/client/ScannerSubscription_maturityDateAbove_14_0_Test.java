package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateAbove_14_0_Test {

    @Test
    void testMaturityDateAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with no value set
        assertNull(subscription.maturityDateAbove(), "Should return null when no value is set");
        // Test with a value set
        String expectedMaturityDate = "2024-12-31";
        subscription.maturityDateAbove(expectedMaturityDate);
        assertEquals(expectedMaturityDate, subscription.maturityDateAbove(), "Should return the set value");
        // Test with null value
        subscription.maturityDateAbove(null);
        assertNull(subscription.maturityDateAbove(), "Should return null when null value is set");
        // Test with empty string
        subscription.maturityDateAbove("");
        assertEquals("", subscription.maturityDateAbove(), "Should return empty string when empty string is set");
    }
}
