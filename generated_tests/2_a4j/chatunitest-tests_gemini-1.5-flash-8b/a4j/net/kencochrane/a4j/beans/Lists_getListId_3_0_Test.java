package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Lists_getListId_3_0_Test {

    @Test
    void testGetListIdNegativeIndex() {
        // Arrange
        ArrayList<String> lists = new ArrayList<>(Arrays.asList("list1", "list2"));
        Lists listObj = new Lists();
        listObj.setListId(lists.toArray(new String[0]));
        // Act
        String result = listObj.getListId(-1);
        // Assert
        assertNull(result);
    }
}
