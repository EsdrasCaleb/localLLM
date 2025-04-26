package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_17_1_Test {

    @Test
    public void testCouponRateBelow() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.instrument("Instrument1");
        scannerSubscription.locationCode("Location1");
        scannerSubscription.scanCode("ScanCode1");
        scannerSubscription.abovePrice(10.0);
        scannerSubscription.belowPrice(20.0);
        scannerSubscription.averageOptionVolumeAbove(100);
        scannerSubscription.marketCapAbove(1000.0);
        scannerSubscription.moodyRatingAbove("MoodyRating1");
        scannerSubscription.spRatingAbove("SpRating1");
        scannerSubscription.maturityDateAbove("MaturityDate1");
        scannerSubscription.couponRateAbove(5.0);
        scannerSubscription.scannerSettingPairs("ScannerSettingPairs1");
        scannerSubscription.stockTypeFilter("StockTypeFilter1");
        double couponRateBelow = scannerSubscription.couponRateBelow();
        assertEquals(Double.MAX_VALUE, couponRateBelow, 0.0);
    }

    @Test
    public void testCouponRateBelow_NoCouponRate() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.abovePrice(10.0);
        scannerSubscription.belowPrice(20.0);
        scannerSubscription.averageOptionVolumeAbove(100);
        scannerSubscription.marketCapAbove(1000.0);
        scannerSubscription.moodyRatingAbove("MoodyRating1");
        scannerSubscription.spRatingAbove("SpRating1");
        scannerSubscription.maturityDateAbove("MaturityDate1");
        scannerSubscription.couponRateAbove(5.0);
        scannerSubscription.scannerSettingPairs("ScannerSettingPairs1");
        scannerSubscription.stockTypeFilter("StockTypeFilter1");
        double couponRateBelow = scannerSubscription.couponRateBelow();
        assertEquals(Double.MAX_VALUE, couponRateBelow, 0.0);
    }

    @Test
    public void testCouponRateBelow_NegativeCouponRate() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.abovePrice(10.0);
        scannerSubscription.belowPrice(20.0);
        scannerSubscription.averageOptionVolumeAbove(100);
        scannerSubscription.marketCapAbove(1000.0);
        scannerSubscription.moodyRatingAbove("MoodyRating1");
        scannerSubscription.spRatingAbove("SpRating1");
        scannerSubscription.maturityDateAbove("MaturityDate1");
        scannerSubscription.couponRateAbove(-5.0);
        scannerSubscription.scannerSettingPairs("ScannerSettingPairs1");
        scannerSubscription.stockTypeFilter("StockTypeFilter1");
        double couponRateBelow = scannerSubscription.couponRateBelow();
        assertEquals(Double.MAX_VALUE, couponRateBelow, 0.0);
    }
}
