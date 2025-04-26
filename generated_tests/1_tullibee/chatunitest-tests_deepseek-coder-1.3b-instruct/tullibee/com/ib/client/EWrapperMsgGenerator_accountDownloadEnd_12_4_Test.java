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

class EWrapperMsgGenerator_accountDownloadEnd_12_4_Test {

    @Test
    void accountDownloadEndTest() {
        String expected = "accountDownloadEnd: testAccount";
        String actual = EWrapperMsgGenerator.accountDownloadEnd("testAccount");
        assertEquals(expected, actual);
    }
}
