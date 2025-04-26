package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingAbove_12_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testSpRatingAbove_WhenNotSet_ShouldReturnNull() {
        // Act
        String result = scannerSubscription.spRatingAbove();
        // Assert
        assertEquals(null, result);
    }

    @Test
    public void testSpRatingAbove_WhenSet_ShouldReturnValue() {
        // Arrange
        String expectedRating = "A+";
        scannerSubscription.spRatingAbove(expectedRating);
        // Act
        String result = scannerSubscription.spRatingAbove();
        // Assert
        assertEquals(expectedRating, result);
    }

    @Test
    public void testSpRatingAbove_WhenSetToEmptyString_ShouldReturnEmptyString() {
        // Arrange
        String expectedRating = "";
        scannerSubscription.spRatingAbove(expectedRating);
        // Act
        String result = scannerSubscription.spRatingAbove();
        // Assert
        assertEquals(expectedRating, result);
    }

    @Test
    public void testSpRatingAbove_WhenSetToNull_ShouldReturnNull() {
        // Arrange
        scannerSubscription.spRatingAbove(null);
        // Act
        String result = scannerSubscription.spRatingAbove();
        // Assert
        assertEquals(null, result);
    }
}
