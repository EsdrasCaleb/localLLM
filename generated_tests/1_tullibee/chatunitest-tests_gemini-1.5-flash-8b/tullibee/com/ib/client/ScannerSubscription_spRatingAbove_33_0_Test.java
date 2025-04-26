package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_spRatingAbove_33_0_Test {

    private ScannerSubscription subscription;

    @BeforeEach
    void setUp() {
        subscription = new ScannerSubscription();
    }

    @ParameterizedTest
    @ValueSource(strings = { "AAA", "BBB", "AA+", "A-", "null" })
    void testSpRatingAbove(String rating) throws NoSuchFieldException, IllegalAccessException {
        // Use reflection to set the private field
        try {
            Field spRatingAboveField = ScannerSubscription.class.getDeclaredField("m_spRatingAbove");
            spRatingAboveField.setAccessible(true);
            subscription.spRatingAbove(rating);
            String actualRating = (String) spRatingAboveField.get(subscription);
            // Handle null case correctly
            if (rating == null) {
                assertNull(actualRating);
            } else {
                assertEquals(rating, actualRating);
            }
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException("Error accessing private field", e);
        }
    }
}
