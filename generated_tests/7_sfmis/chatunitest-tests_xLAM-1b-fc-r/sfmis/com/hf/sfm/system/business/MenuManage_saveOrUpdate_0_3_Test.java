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

public class MenuManage_saveOrUpdate_0_3_Test {

    @Mock
    private MenuManage menuManage;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testSaveOrUpdate() {
        // Given
        Menu menu = new Menu();
        menu.setIdno("123");
        menu.setName("Test Menu");
        menu.setImg("test.jpg");
        menu.setStatus("Active");
        // When
        when(menuManage.saveOrUpdate(menu)).thenReturn("1");
        // Then
        String result = menuManage.saveOrUpdate(menu);
        assertEquals("1", result);
    }
}
