package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingAbove_10_0_Test {

    @Test
    void testMoodyRatingAbove_NoValueSet() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertNull(subscription.moodyRatingAbove(), "Should return null when no value is set");
    }

    @Test
    void testMoodyRatingAbove_ValueSet() {
        ScannerSubscription subscription = new ScannerSubscription();
        String expectedRating = "Aaa";
        subscription.moodyRatingAbove(expectedRating);
        assertEquals(expectedRating, subscription.moodyRatingAbove(), "Should return the set value");
    }

    @Test
    void testMoodyRatingAbove_ValueSetThenCleared() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.moodyRatingAbove("Aaa");
        subscription.moodyRatingAbove(null);
        assertNull(subscription.moodyRatingAbove(), "Should return null after clearing the value");
    }

    @Test
    void testMoodyRatingAbove_EmptyString() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.moodyRatingAbove("");
        assertEquals("", subscription.moodyRatingAbove(), "Should return empty string when empty string is set");
    }
}
