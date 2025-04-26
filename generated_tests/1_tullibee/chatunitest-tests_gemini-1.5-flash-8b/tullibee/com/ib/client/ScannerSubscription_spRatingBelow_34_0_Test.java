package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_spRatingBelow_34_0_Test {

    @Test
    void testSpRatingBelow() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with a valid input string
        String rating = "BBB";
        subscription.spRatingBelow(rating);
        Field spRatingBelowField = ScannerSubscription.class.getDeclaredField("m_spRatingBelow");
        spRatingBelowField.setAccessible(true);
        String actualRating = (String) spRatingBelowField.get(subscription);
        Assertions.assertEquals(rating, actualRating);
        // Test with null input
        rating = null;
        subscription.spRatingBelow(rating);
        actualRating = (String) spRatingBelowField.get(subscription);
        Assertions.assertNull(actualRating);
        // Test with an empty string
        rating = "";
        subscription.spRatingBelow(rating);
        actualRating = (String) spRatingBelowField.get(subscription);
        Assertions.assertEquals("", actualRating);
    }
}
