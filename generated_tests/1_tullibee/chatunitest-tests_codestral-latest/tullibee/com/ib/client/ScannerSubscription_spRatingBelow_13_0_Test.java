package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_spRatingBelow_13_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testSpRatingBelow_DefaultValue() {
        assertNull(scannerSubscription.spRatingBelow());
    }

    @Test
    public void testSpRatingBelow_SetValue() {
        String expectedRating = "AAA";
        scannerSubscription.spRatingBelow(expectedRating);
        assertEquals(expectedRating, scannerSubscription.spRatingBelow());
    }
}
