package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Accessories_getAccessory_3_3_Test {

    @Mock
    private Accessories accessor;

    @InjectMocks
    private Accessories accessorMock;

    @Test
    public void testGetAccessory() {
        // Arrange
        List<String> expected = new ArrayList<>();
        expected.add("Accessory 1");
        expected.add("Accessory 2");
        expected.add("Accessory 3");
        // Act
        String result = accessor.getAccessory(0);
        // Assert
        assertEquals(expected, result);
    }

    @Test
    public void testGetAccessory_1() {
        // Arrange
        List<String> expected = new ArrayList<>();
        expected.add("Accessory 1");
        expected.add("Accessory 2");
        expected.add("Accessory 3");
        // Act
        String result = accessor.getAccessory(0);
        // Assert
        assertEquals(expected, result);
    }

    @Test
    public void testGetAccessory_2() {
        // Arrange
        List<String> expected = new ArrayList<>();
        expected.add("Accessory 1");
        expected.add("Accessory 2");
        expected.add("Accessory 3");
        // Act
        String result = accessor.getAccessory(1);
        // Assert
        assertEquals(expected, result);
    }

    @Test
    public void testGetAccessory_3() {
        // Arrange
        List<String> expected = new ArrayList<>();
        expected.add("Accessory 1");
        expected.add("Accessory 2");
        expected.add("Accessory 3");
        // Act
        String result = accessor.getAccessory(2);
        // Assert
        assertEquals(expected, result);
    }

    @Test
    public void testGetAccessory_4() {
        // Arrange
        List<String> expected = new ArrayList<>();
        expected.add("Accessory 1");
        expected.add("Accessory 2");
        expected.add("Accessory 3");
        // Act
        String result = accessor.getAccessory(4);
        // Assert
        assertEquals(expected, result);
    }

    @Test
    public void testGetAccessory_5() {
        // Arrange
        List<String> expected = new ArrayList<>();
        expected.add("Accessory 1");
        expected.add("Accessory 2");
        expected.add("Accessory 3");
        // Act
        String result = accessor.getAccessory(5);
        // Assert
        assertEquals(expected, result);
    }

    @Test
    public void testGetAccessory_6() {
        // Arrange
        List<String> expected = new ArrayList<>();
        expected.add("Accessory 1");
        expected.add("Accessory 2");
        expected.add("Accessory 3");
        // Act
        String result = accessor.getAccessory(6);
        // Assert
        assertEquals(expected, result);
    }

    @Test
    public void testGetAccessory_7() {
        // Arrange
        List<String> expected = new ArrayList<>();
        expected.add("Accessory 1");
        expected.add("Accessory 2");
        expected.add("Accessory 3");
        // Act
        String result = accessor.getAccessory(7);
        // Assert
        assertEquals(expected, result);
    }

    @Test
    public void testGetAccessory_8() {
        // Arrange
        List<String> expected = new ArrayList<>();
        expected.add("Accessory 1");
        expected.add("Accessory 2");
        expected.add("Accessory 3");
        // Act
        String result = accessor.getAccessory(8);
        // Assert
        assertEquals(expected, result);
    }

    @Test
    public void testGetAccessory_9() {
        // Arrange
        List<String> expected = new ArrayList<>();
        expected.add("Accessory 1");
        expected.add("Accessory 2");
        expected.add("Accessory 3");
        // Act
        String result = accessor.getAccessory(9);
        // Assert
        assertEquals(expected, result);
    }
}
