package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingAbove_31_1_Test {

    @Test
    public void testMoodyRatingAbove_SettingMoodyRatingAbove() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.moodyRatingAbove("Baa3");
        assertEquals("Baa3", scannerSubscription.moodyRatingAbove());
    }

    @Test
    public void testMoodyRatingAbove_SettingMoodyRatingBelow() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.moodyRatingBelow("Baa3");
        assertEquals("Baa3", scannerSubscription.moodyRatingBelow());
    }

    @Test
    public void testMoodyRatingAbove_InvalidInput() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        assertThrows(NullPointerException.class, () -> scannerSubscription.moodyRatingAbove(null));
    }

    @Test
    public void testMoodyRatingAbove_EmptyString() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.moodyRatingAbove("");
        assertEquals("", scannerSubscription.moodyRatingAbove());
    }

    @Test
    public void testMoodyRatingAbove_SingleCharacter() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.moodyRatingAbove("A");
        assertEquals("A", scannerSubscription.moodyRatingAbove());
    }

    @Test
    public void testMoodyRatingAbove_MultipleCharacters() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.moodyRatingAbove("Aa");
        assertEquals("Aa", scannerSubscription.moodyRatingAbove());
    }
}
