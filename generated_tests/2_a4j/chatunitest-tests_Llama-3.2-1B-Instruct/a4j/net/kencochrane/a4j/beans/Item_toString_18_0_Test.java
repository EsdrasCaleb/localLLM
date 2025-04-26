package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.HashMap;
import java.util.Map;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Item_toString_18_0_Test {

    @Mock
    private Item item;

    @InjectMocks
    private Item itemToTest;

    @Test
    public void testToString() {
        // Arrange
        Map<String, Object> expectedOutput = new HashMap<>();
        expectedOutput.put("asin", "asin");
        expectedOutput.put("name", "name");
        expectedOutput.put("quantity", "quantity");
        expectedOutput.put("listPrice", "listPrice");
        expectedOutput.put("ourPrice", "ourPrice");
        expectedOutput.put("merchantSku", "merchantSku");
        // Act
        String output = item.toString();
        // Assert
        assertEquals(expectedOutput, output);
    }
}
