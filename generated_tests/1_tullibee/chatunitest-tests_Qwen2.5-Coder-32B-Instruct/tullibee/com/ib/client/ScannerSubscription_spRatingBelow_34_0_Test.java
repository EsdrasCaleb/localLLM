package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingBelow_34_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testSpRatingBelow() throws NoSuchFieldException, IllegalAccessException {
        // Given
        String testRating = "BBB";
        // When
        scannerSubscription.spRatingBelow(testRating);
        // Then
        Field spRatingBelowField = ScannerSubscription.class.getDeclaredField("m_spRatingBelow");
        spRatingBelowField.setAccessible(true);
        String actualRating = (String) spRatingBelowField.get(scannerSubscription);
        assertEquals(testRating, actualRating);
    }
}
