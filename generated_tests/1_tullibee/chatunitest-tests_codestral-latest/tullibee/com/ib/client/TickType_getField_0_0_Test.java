package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class TickType_getField_0_0_Test {

    @Test
    public void testGetField() {
        assertEquals("bidSize", TickType.getField(TickType.BID_SIZE));
        assertEquals("bidPrice", TickType.getField(TickType.BID));
        assertEquals("askPrice", TickType.getField(TickType.ASK));
        assertEquals("askSize", TickType.getField(TickType.ASK_SIZE));
        assertEquals("lastPrice", TickType.getField(TickType.LAST));
        assertEquals("lastSize", TickType.getField(TickType.LAST_SIZE));
        assertEquals("high", TickType.getField(TickType.HIGH));
        assertEquals("low", TickType.getField(TickType.LOW));
        assertEquals("volume", TickType.getField(TickType.VOLUME));
        assertEquals("close", TickType.getField(TickType.CLOSE));
        assertEquals("bidOptComp", TickType.getField(TickType.BID_OPTION));
        assertEquals("askOptComp", TickType.getField(TickType.ASK_OPTION));
        assertEquals("lastOptComp", TickType.getField(TickType.LAST_OPTION));
        assertEquals("modelOptComp", TickType.getField(TickType.MODEL_OPTION));
        assertEquals("open", TickType.getField(TickType.OPEN));
        assertEquals("13WeekLow", TickType.getField(TickType.LOW_13_WEEK));
        assertEquals("13WeekHigh", TickType.getField(TickType.HIGH_13_WEEK));
        assertEquals("26WeekLow", TickType.getField(TickType.LOW_26_WEEK));
        assertEquals("26WeekHigh", TickType.getField(TickType.HIGH_26_WEEK));
        assertEquals("52WeekLow", TickType.getField(TickType.LOW_52_WEEK));
        assertEquals("52WeekHigh", TickType.getField(TickType.HIGH_52_WEEK));
        assertEquals("AvgVolume", TickType.getField(TickType.AVG_VOLUME));
        assertEquals("OpenInterest", TickType.getField(TickType.OPEN_INTEREST));
        assertEquals("OptionHistoricalVolatility", TickType.getField(TickType.OPTION_HISTORICAL_VOL));
        assertEquals("OptionImpliedVolatility", TickType.getField(TickType.OPTION_IMPLIED_VOL));
        assertEquals("OptionBidExchStr", TickType.getField(TickType.OPTION_BID_EXCH));
        assertEquals("OptionAskExchStr", TickType.getField(TickType.OPTION_ASK_EXCH));
        assertEquals("OptionCallOpenInterest", TickType.getField(TickType.OPTION_CALL_OPEN_INTEREST));
        assertEquals("OptionPutOpenInterest", TickType.getField(TickType.OPTION_PUT_OPEN_INTEREST));
        assertEquals("OptionCallVolume", TickType.getField(TickType.OPTION_CALL_VOLUME));
        assertEquals("OptionPutVolume", TickType.getField(TickType.OPTION_PUT_VOLUME));
        assertEquals("IndexFuturePremium", TickType.getField(TickType.INDEX_FUTURE_PREMIUM));
        assertEquals("bidExch", TickType.getField(TickType.BID_EXCH));
        assertEquals("askExch", TickType.getField(TickType.ASK_EXCH));
        assertEquals("auctionVolume", TickType.getField(TickType.AUCTION_VOLUME));
        assertEquals("auctionPrice", TickType.getField(TickType.AUCTION_PRICE));
        assertEquals("auctionImbalance", TickType.getField(TickType.AUCTION_IMBALANCE));
        assertEquals("markPrice", TickType.getField(TickType.MARK_PRICE));
        assertEquals("bidEFP", TickType.getField(TickType.BID_EFP_COMPUTATION));
        assertEquals("askEFP", TickType.getField(TickType.ASK_EFP_COMPUTATION));
        assertEquals("lastEFP", TickType.getField(TickType.LAST_EFP_COMPUTATION));
        assertEquals("openEFP", TickType.getField(TickType.OPEN_EFP_COMPUTATION));
        assertEquals("highEFP", TickType.getField(TickType.HIGH_EFP_COMPUTATION));
        assertEquals("lowEFP", TickType.getField(TickType.LOW_EFP_COMPUTATION));
    }
}
