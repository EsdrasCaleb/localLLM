package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingAbove_12_0_Test {

    @Test
    void testSpRatingAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with initial value (null)
        assertNull(subscription.spRatingAbove(), "Initial value should be null");
        // Test after setting a value
        subscription.spRatingAbove("BBB+");
        assertEquals("BBB+", subscription.spRatingAbove(), "Value should be correctly set and retrieved");
        // Test setting to null
        subscription.spRatingAbove(null);
        assertNull(subscription.spRatingAbove(), "Value should be correctly set to null");
        // Test setting to an empty string
        subscription.spRatingAbove("");
        assertEquals("", subscription.spRatingAbove(), "Value should be correctly set to an empty string");
        // Test setting to a different value
        subscription.spRatingAbove("AA-");
        assertEquals("AA-", subscription.spRatingAbove(), "Value should be correctly set and retrieved");
    }
}
