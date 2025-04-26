package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingBelow_11_0_Test {

    @Test
    void testMoodyRatingBelow() {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with default value
        assertNull(subscription.moodyRatingBelow(), "Default value should be null");
        // Test with a set value
        String expectedRating = "Baa1";
        subscription.moodyRatingBelow(expectedRating);
        assertEquals(expectedRating, subscription.moodyRatingBelow(), "Set value should be returned");
        // Test with null value
        subscription.moodyRatingBelow(null);
        assertNull(subscription.moodyRatingBelow(), "Null value should be returned");
        // Test with an empty string
        subscription.moodyRatingBelow("");
        assertEquals("", subscription.moodyRatingBelow(), "Empty string should be returned");
    }
}
