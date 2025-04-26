package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_contractMsg_16_0_Test {

    @Mock
    private Contract contract;

    @InjectMocks
    private EWrapperMsgGenerator focal;

    @Test
    @ParameterizedTest
    @CsvSource({ "CONID=1234,SEC_TYPE=FA,EXPIRY=2022-01-01,STRIKE=100,RIGHT=1,MULTIPLIER=2,EXCHANGE=USD,PRIMARY_EXCH=NYSE,LOCAL_SYMBOL=EUR,LOCAL.exchange=USD,CURRENCY=EUR,LOCALSymbol=EUR", "CONID=5678,SEC_TYPE=FA,EXPIRY=2022-01-02,STRIKE=200,RIGHT=2,MULTIPLIER=3,EXCHANGE=USD,PRIMARY_EXCH=NYSE,LOCAL_SYMBOL=USD,LOCAL.exchange=USD,CURRENCY=USD,LOCALSymbol=USD" })
    public void testContractMsg() {
        String msg = focal.contractMsg(contract);
        assertEquals("CONID = 1234\n" + "SEC_TYPE = FA\n" + "EXPIRY = 2022-01-01\n" + "STRIKE = 100\n" + "RIGHT = 1\n" + "MULTIPLIER = 2\n" + "EXCHANGE = USD\n" + "PRIMARY_EXCH = NYSE\n" + "LOCAL_SYMBOL = EUR\n" + "LOCAL.exchange = USD\n" + "CURRENCY = USD\n" + "LOCALSymbol = EUR", msg);
    }
}
