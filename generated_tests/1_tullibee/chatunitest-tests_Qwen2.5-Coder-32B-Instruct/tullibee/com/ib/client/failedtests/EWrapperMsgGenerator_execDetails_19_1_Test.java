package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_execDetails_19_1_Test {

    @Mock
    private Contract mockContract;

    @Mock
    private Execution mockExecution;

    private EWrapperMsgGenerator eWrapperMsgGenerator;

    private static final String FIELD_SEP = "\t";

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        eWrapperMsgGenerator = new EWrapperMsgGenerator();
        // Setting up the mock objects with expected values
        when(mockContract.m_symbol).thenReturn("AAPL");
        when(mockContract.m_secType).thenReturn("STK");
        when(mockContract.m_expiry).thenReturn("20231020");
        when(mockContract.m_strike).thenReturn(150.0);
        when(mockContract.m_right).thenReturn("CALL");
        when(mockContract.m_exchange).thenReturn("NASDAQ");
        when(mockContract.m_currency).thenReturn("USD");
        when(mockContract.m_localSymbol).thenReturn("AAPL231020C00150000");
        when(mockExecution.m_orderId).thenReturn(12345);
        when(mockExecution.m_clientId).thenReturn(67890);
        when(mockExecution.m_execId).thenReturn("123456789");
        when(mockExecution.m_time).thenReturn("20231010 10:00:00");
        when(mockExecution.m_acctNumber).thenReturn("U123456789");
        when(mockExecution.m_exchange).thenReturn("SMART");
        when(mockExecution.m_side).thenReturn("BUY");
        when(mockExecution.m_shares).thenReturn(100);
        when(mockExecution.m_price).thenReturn(149.5);
        when(mockExecution.m_permId).thenReturn(98765);
        when(mockExecution.m_liquidation).thenReturn(0);
        when(mockExecution.m_cumQty).thenReturn(100);
        when(mockExecution.m_avgPrice).thenReturn(149.5);
        // Accessing the private FIELD_SEP field from EWrapperMsgGenerator
        Field field = EWrapperMsgGenerator.class.getDeclaredField("FIELD_SEP");
        field.setAccessible(true);
        // No need to reassign FIELD_SEP as it's already defined as a static final in the class
    }

    @Test
    public void testExecDetails() {
        StringBuilder expected = new StringBuilder();
        expected.append(19).append(FIELD_SEP);
        expected.append(12345).append(FIELD_SEP);
        expected.append(67890).append(FIELD_SEP);
        expected.append("123456789").append(FIELD_SEP);
        expected.append("20231010 10:00:00").append(FIELD_SEP);
        expected.append("AAPL").append(FIELD_SEP);
        expected.append("STK").append(FIELD_SEP);
        expected.append("20231020").append(FIELD_SEP);
        expected.append(150.0).append(FIELD_SEP);
        expected.append("CALL").append(FIELD_SEP);
        expected.append("SMART").append(FIELD_SEP);
        expected.append("USD").append(FIELD_SEP);
        expected.append("BUY").append(FIELD_SEP);
        expected.append(100).append(FIELD_SEP);
        expected.append(149.5).append(FIELD_SEP);
        expected.append(98765).append(FIELD_SEP);
        expected.append(0).append(FIELD_SEP);
        expected.append(100).append(FIELD_SEP);
        expected.append(149.5).append(FIELD_SEP);
        expected.append(0).append(FIELD_SEP);
        expected.append(0).append(FIELD_SEP);
        expected.append(0).append(FIELD_SEP);
        expected.append("U123456789").append(FIELD_SEP);
        expected.append("AAPL231020C00150000").append(FIELD_SEP);
        // Removed the incorrect line and added the missing fields
        // m_lastLiquidity
        expected.append(0).append(FIELD_SEP);
        // m_clientTag
        expected.append("").append(FIELD_SEP);
        // m_attrLastPrice
        expected.append("").append(FIELD_SEP);
        // m_attrLastPriceTranType
        expected.append("").append(FIELD_SEP);
        expected.append("").append(FIELD_SEP);
    }
}
