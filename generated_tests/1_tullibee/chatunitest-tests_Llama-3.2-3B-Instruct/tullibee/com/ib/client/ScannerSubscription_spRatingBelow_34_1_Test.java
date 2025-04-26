// Test method
package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_spRatingBelow_34_1_Test {

    @Mock
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testSpRatingBelow() {
        // Arrange
        String input = "Rating1";
        when(scannerSubscription.spRatingBelow()).thenReturn(input);
        String actual = scannerSubscription.spRatingBelow();
        // Assert
        assertEquals(input, actual);
    }

    @Test
    public void testSpRatingBelow_EmptyString() {
        // Arrange
        when(scannerSubscription.spRatingBelow()).thenReturn("");
        String actual = scannerSubscription.spRatingBelow();
        // Assert
        assertEquals("", actual);
    }

    @Test
    public void testSpRatingBelow_NullString() {
        // Arrange
        when(scannerSubscription.spRatingBelow()).thenReturn(null);
        String actual = scannerSubscription.spRatingBelow();
        // Assert
        assertNull(actual);
    }
}
