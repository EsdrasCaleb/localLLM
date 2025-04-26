package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingAbove_12_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testSpRatingAbove() throws NoSuchFieldException, IllegalAccessException {
        // Set up the private field m_spRatingAbove using reflection
        Field spRatingAboveField = ScannerSubscription.class.getDeclaredField("m_spRatingAbove");
        spRatingAboveField.setAccessible(true);
        // Test case 1: m_spRatingAbove is set to a non-null value
        String expectedRating = "AAA";
        spRatingAboveField.set(scannerSubscription, expectedRating);
        String actualRating = scannerSubscription.spRatingAbove();
        assertEquals(expectedRating, actualRating);
        // Test case 2: m_spRatingAbove is set to null
        spRatingAboveField.set(scannerSubscription, null);
        actualRating = scannerSubscription.spRatingAbove();
        assertNull(actualRating);
    }
}
