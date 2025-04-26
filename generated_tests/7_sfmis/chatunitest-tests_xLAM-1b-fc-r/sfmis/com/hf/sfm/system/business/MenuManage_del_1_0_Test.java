package com.hf.sfm.system.business;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.hibernate.Transaction;
import com.hf.sfm.system.pdo.Menu;
import com.hf.sfm.util.DaoFactory;

public class MenuManage_del_1_0_Test {

    @Mock
    MenuManage menuManage;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    void testDel() {
        // Given
        String[] idnos = { "123", "456" };
        String expectedResult = "1";
        // When
        when(menuManage.del(idnos)).thenReturn(expectedResult);
        // Then
        String actualResult = menuManage.del(idnos);
        assertEquals(expectedResult, actualResult);
    }
}
