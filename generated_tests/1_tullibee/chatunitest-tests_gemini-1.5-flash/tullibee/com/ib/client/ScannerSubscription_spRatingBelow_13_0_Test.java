package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingBelow_13_0_Test {

    @Test
    void testSpRatingBelow() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with default value
        assertEquals(null, subscription.spRatingBelow());
        // Test with a set value
        subscription.spRatingBelow("BBB-");
        assertEquals("BBB-", subscription.spRatingBelow());
        // Test with null value
        subscription.spRatingBelow(null);
        assertEquals(null, subscription.spRatingBelow());
        // Test with empty string
        subscription.spRatingBelow("");
        assertEquals("", subscription.spRatingBelow());
    }
}
