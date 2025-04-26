package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import java.util.stream.Stream;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_orderStatus_6_0_Test {

    @ParameterizedTest
    @MethodSource("orderStatusProvider")
    void orderStatusTest(int orderId, String status, int filled, int remaining, double avgFillPrice, int permId, int parentId, double lastFillPrice, int clientId, String whyHeld, String expectedOutput) {
        String actualOutput = EWrapperMsgGenerator.orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld);
        assertEquals(expectedOutput, actualOutput);
    }

    static Stream<Arguments> orderStatusProvider() {
        return Stream.of(Arguments.of(1, "Filled", 10, 0, 10.50, 123, 0, 10.50, 456, null, "order status: orderId=1 clientId=456 permId=123 status=Filled filled=10 remaining=0 avgFillPrice=10.5 lastFillPrice=10.5 parent Id=0 whyHeld=null"), Arguments.of(2, "Pending", 5, 5, 12.75, 456, 1, 12.75, 789, "Limit order exceeded", "order status: orderId=2 clientId=789 permId=456 status=Pending filled=5 remaining=5 avgFillPrice=12.75 lastFillPrice=12.75 parent Id=1 whyHeld=Limit order exceeded"), Arguments.of(3, "Cancelled", 0, 10, 0, 789, 0, 0, 123, null, "order status: orderId=3 clientId=123 permId=789 status=Cancelled filled=0 remaining=10 avgFillPrice=0.0 lastFillPrice=0.0 parent Id=0 whyHeld=null"), Arguments.of(4, "Rejected", 0, 0, 0, 123, 0, 0, 456, "Invalid price", "order status: orderId=4 clientId=456 permId=123 status=Rejected filled=0 remaining=0 avgFillPrice=0.0 lastFillPrice=0.0 parent Id=0 whyHeld=Invalid price"));
    }
}
