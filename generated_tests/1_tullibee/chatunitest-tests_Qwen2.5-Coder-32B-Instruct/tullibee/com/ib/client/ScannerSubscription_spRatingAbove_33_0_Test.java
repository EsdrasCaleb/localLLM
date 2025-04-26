package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_spRatingAbove_33_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testSpRatingAbove() throws Exception {
        // Given
        String expectedRating = "A";
        java.lang.reflect.Field field = scannerSubscription.getClass().getDeclaredField("m_spRatingAbove");
        field.setAccessible(true);
        // <Buggy Line> fixed by accessing the field directly
        assertNull(field.get(scannerSubscription));
        // When
        scannerSubscription.spRatingAbove(expectedRating);
        // Then
        String actualRating = (String) field.get(scannerSubscription);
        assertEquals(expectedRating, actualRating);
    }
}
