package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingAbove_10_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testMoodyRatingAbove() {
        // Set up the test data
        String expectedMoodyRatingAbove = "A1";
        scannerSubscription.moodyRatingAbove(expectedMoodyRatingAbove);
        // Invoke the method under test
        String actualMoodyRatingAbove = scannerSubscription.moodyRatingAbove();
        // Verify the result
        assertEquals(expectedMoodyRatingAbove, actualMoodyRatingAbove, "The moodyRatingAbove value should match the expected value.");
    }
}
