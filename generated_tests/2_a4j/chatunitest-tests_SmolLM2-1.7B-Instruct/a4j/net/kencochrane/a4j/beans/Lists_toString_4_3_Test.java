package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Lists_toString_4_3_Test {

    @Mock
    private Lists lists;

    @InjectMocks
    private Lists focal;

    @Test
    public void testToString() {
        // Arrange
        focal.lists = new ArrayList<>();
        focal.lists.add("list1");
        focal.lists.add("list2");
        // Act
        String output = focal.toString();
        // Assert
        assert output.contains("# of Lists = 2");
        assert output.contains("list - list1");
        assert output.contains("list - list2");
    }
}
