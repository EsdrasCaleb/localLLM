package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingBelow_32_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMoodyRatingBelow() throws NoSuchFieldException, IllegalAccessException {
        // Given
        String testRating = "Baa1";
        scannerSubscription.moodyRatingBelow(testRating);
        // When
        Field moodyRatingBelowField = ScannerSubscription.class.getDeclaredField("m_moodyRatingBelow");
        moodyRatingBelowField.setAccessible(true);
        String result = (String) moodyRatingBelowField.get(scannerSubscription);
        // Then
        assertEquals(testRating, result, "The moodyRatingBelow should be set to the provided value.");
    }
}
