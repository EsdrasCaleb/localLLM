package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateBelow_15_0_Test {

    @Test
    void testMaturityDateBelow() {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with default value
        assertNull(subscription.maturityDateBelow(), "Default value should be null");
        // Test with a set value
        String expectedMaturityDate = "2024-12-31";
        subscription.maturityDateBelow(expectedMaturityDate);
        assertEquals(expectedMaturityDate, subscription.maturityDateBelow(), "Set value should be returned");
        // Test with null value
        subscription.maturityDateBelow(null);
        assertNull(subscription.maturityDateBelow(), "Null value should be returned");
        // Test with empty string
        subscription.maturityDateBelow("");
        assertEquals("", subscription.maturityDateBelow(), "Empty string should be returned");
    }
}
