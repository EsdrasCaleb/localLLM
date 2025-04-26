package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class ScannerSubscription_moodyRatingBelow_32_1_Test {

    @Test
    public void testMoodyRatingBelow() {
        ScannerSubscription subscription = new ScannerSubscription();
        String expectedRating = "High";
        subscription.moodyRatingBelow(expectedRating);
        assertEquals(expectedRating, subscription.moodyRatingBelow());
    }
}
