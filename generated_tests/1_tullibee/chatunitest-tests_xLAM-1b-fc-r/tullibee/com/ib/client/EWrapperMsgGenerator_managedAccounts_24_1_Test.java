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

public class EWrapperMsgGenerator_managedAccounts_24_1_Test {

    @Test
    public void testManagedAccounts() {
        // Test with a valid list of managed accounts
        String expected = "Connected : The list of managed accounts are : [account1, account2, account3]";
        String result = EWrapperMsgGenerator.managedAccounts("account1, account2, account3");
        assertEquals(expected, result);
        // Test with an empty list of managed accounts
        expected = "Connected : The list of managed accounts are : []";
        result = EWrapperMsgGenerator.managedAccounts("");
        assertEquals(expected, result);
        // Test with a null list of managed accounts
        result = EWrapperMsgGenerator.managedAccounts(null);
        assertEquals(expected, result);
    }
}
