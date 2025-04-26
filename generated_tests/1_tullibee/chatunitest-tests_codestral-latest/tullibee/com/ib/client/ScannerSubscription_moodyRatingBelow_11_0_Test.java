package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingBelow_11_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testMoodyRatingBelow() throws Exception {
        // Set up the private field using reflection
        String expectedRating = "A1";
        java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField("m_moodyRatingBelow");
        field.setAccessible(true);
        field.set(scannerSubscription, expectedRating);
        // Invoke the method and verify the result
        String actualRating = scannerSubscription.moodyRatingBelow();
        assertEquals(expectedRating, actualRating);
    }
}
