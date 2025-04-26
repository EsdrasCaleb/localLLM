package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class ScannerSubscription_spRatingBelow_34_1_Test {

    @Mock
    ScannerSubscription subscription;

    @Test
    void spRatingBelow_setsCorrectValue() {
        String rating = "Test Rating";
        doNothing().when(subscription).spRatingBelow(rating);
        subscription.spRatingBelow(rating);
        verify(subscription, times(1)).spRatingBelow(rating);
    }
}
