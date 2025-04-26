package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_accountDownloadEnd_12_1_Test {

    // Unit test
    @Test
    public void testAccountDownloadEnd() {
        String accountName = "123456";
        String expected = "accountDownloadEnd: " + accountName;
        String actual = EWrapperMsgGenerator.accountDownloadEnd(accountName);
        assertEquals(expected, actual);
    }
}
