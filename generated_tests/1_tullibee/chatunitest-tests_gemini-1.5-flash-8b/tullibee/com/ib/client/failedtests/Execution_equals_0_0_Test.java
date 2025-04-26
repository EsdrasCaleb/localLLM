package com.ib.client;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Execution_equals_0_0_Test {

    @Test
    void equals_nullObject() {
        Execution execution = new Execution(1, 2, "execId1", "2023-10-27 10:00:00", "acct1", "exchange1", "buy", 100, 10.0, 1, 0, 100, 10.0);
        assertFalse(execution.equals(null));
    }

    @Test
    void equals_sameObject() {
        Execution execution = new Execution(1, 2, "execId1", "2023-10-27 10:00:00", "acct1", "exchange1", "buy", 100, 10.0, 1, 0, 100, 10.0);
        assertTrue(execution.equals(execution));
    }

    @Test
    void equals_differentExecId() {
        Execution execution1 = new Execution(1, 2, "execId1", "2023-10-27 10:00:00", "acct1", "exchange1", "buy", 100, 10.0, 1, 0, 100, 10.0);
        Execution execution2 = new Execution(1, 2, "execId2", "2023-10-27 10:00:00", "acct1", "exchange1", "buy", 100, 10.0, 1, 0, 100, 10.0);
        assertFalse(execution1.equals(execution2));
    }

    @Test
    void equals_sameExecId() {
        Execution execution1 = new Execution(1, 2, "execId1", "2023-10-27 10:00:00", "acct1", "exchange1", "buy", 100, 10.0, 1, 0, 100, 10.0);
        Execution execution2 = new Execution(3, 4, "execId1", "2023-10-27 10:00:00", "acct2", "exchange2", "sell", 200, 20.0, 2, 1, 200, 20.0);
        assertTrue(execution1.equals(execution2));
    }

    @Test
    void equals_differentObject() {
        Execution execution = new Execution(1, 2, "execId1", "2023-10-27 10:00:00", "acct1", "exchange1", "buy", 100, 10.0, 1, 0, 100, 10.0);
        assertFalse(execution.equals("not an Execution object"));
    }
}
