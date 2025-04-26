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
    public void testSpRatingBelow_SetValidRating() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedRating = "AA-";
        // Act
        scannerSubscription.spRatingBelow(expectedRating);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_spRatingBelow");
        field.setAccessible(true);
        String actualRating = (String) field.get(scannerSubscription);
        assertEquals(expectedRating, actualRating);
    }

    @Test
    public void testSpRatingBelow_SetNullRating() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedRating = null;
        // Act
        scannerSubscription.spRatingBelow(expectedRating);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_spRatingBelow");
        field.setAccessible(true);
        String actualRating = (String) field.get(scannerSubscription);
        assertNull(actualRating);
    }
}
