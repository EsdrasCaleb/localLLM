package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_spRatingBelow_13_0_Test {

    @Test
    void testSpRatingBelow_validInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        String expectedRating = "BBB";
        try {
            Field spRatingBelowField = ScannerSubscription.class.getDeclaredField("m_spRatingBelow");
            spRatingBelowField.setAccessible(true);
            spRatingBelowField.set(subscription, expectedRating);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        String actualRating = subscription.spRatingBelow();
        assertEquals(expectedRating, actualRating);
    }

    @Test
    void testSpRatingBelow_nullInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        try {
            Field spRatingBelowField = ScannerSubscription.class.getDeclaredField("m_spRatingBelow");
            spRatingBelowField.setAccessible(true);
            spRatingBelowField.set(subscription, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        String actualRating = subscription.spRatingBelow();
        assertNull(actualRating);
    }

    @Test
    void testSpRatingBelow_emptyInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        String expectedRating = "";
        try {
            Field spRatingBelowField = ScannerSubscription.class.getDeclaredField("m_spRatingBelow");
            spRatingBelowField.setAccessible(true);
            spRatingBelowField.set(subscription, expectedRating);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        String actualRating = subscription.spRatingBelow();
        assertEquals(expectedRating, actualRating);
    }
}
