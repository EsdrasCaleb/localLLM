package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Lists_getListId_3_0_Test {

    private Lists lists;

    @BeforeEach
    public void setUp() {
        lists = new Lists();
    }

    @Test
    public void testGetListId_NegativeIndex() throws Exception {
        // Arrange
        String[] newListId = { "item1", "item2", "item3" };
        lists.setListId(newListId);
        // Act
        // Negative index
        String result = lists.getListId(-1);
        // Assert
        assertNull(result);
    }
}
